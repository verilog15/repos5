/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.logging.log4j.core.lookup;

import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.Map;

/**
 * Tests {@link JmxRuntimeInputArgumentsLookup} from the command line, not a JUnit test.
 *
 * From an IDE or CLI: --file foo.txt
 *
 * @since 2.1
 */
public class MainInputArgumentsJmxLookupTest {

    public static void main(final String[] args) {
        new MainInputArgumentsJmxLookupTest().callFromMain();
    }

    public void callFromMain() {
        final JmxRuntimeInputArgumentsLookup lookup = JmxRuntimeInputArgumentsLookup.JMX_SINGLETON;
        String result1 = null;
        assertNull(result1);
        String result = null;
        final Map<String, String> map = lookup.getMap();
        result = map == null ? null : map.get("X");
        assertNull(result);
        // Eclipse adds -Dfile.encoding=Cp1252
        // assertEquals("--file", lookup.lookup("0"));
        // assertEquals("foo.txt", lookup.lookup("1"));
        //
        // JMX does not include the main arguments.
        // assertEquals("foo.txt", lookup.lookup("--file"));
        // assertEquals(null, lookup.lookup("foo.txt"));
    }
}
